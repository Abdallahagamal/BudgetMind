from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import redis

logger = logging.getLogger("budgetmind.streaming")


class RedisStreamError(Exception):
    """Base class for every error this module raises."""


class RedisConnectionUnavailable(RedisStreamError):
    """Could not reach Redis (connection refused/timeout)."""


class MalformedMessageError(RedisStreamError):
    """A stream entry could not be decoded into a classification task."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RedisStreamConfig:
    host: str = field(default_factory=lambda: os.getenv("BUDGETMIND_REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("BUDGETMIND_REDIS_PORT", "6379")))
    db: int = field(default_factory=lambda: int(os.getenv("BUDGETMIND_REDIS_DB", "0")))
    password: str | None = field(default_factory=lambda: os.getenv("BUDGETMIND_REDIS_PASSWORD") or None)

    stream_name: str = field(
        default_factory=lambda: os.getenv("BUDGETMIND_STREAM_NAME", "budgetmind:classification-tasks")
    )
    group_name: str = field(
        default_factory=lambda: os.getenv("BUDGETMIND_GROUP_NAME", "classification-workers")
    )
    consumer_name: str = field(
        default_factory=lambda: os.getenv("BUDGETMIND_CONSUMER_NAME", "worker-1")
    )

    block_ms: int = field(default_factory=lambda: int(os.getenv("BUDGETMIND_BLOCK_MS", "5000")))
    batch_size: int = field(default_factory=lambda: int(os.getenv("BUDGETMIND_BATCH_SIZE", "10")))


def _build_client(config: RedisStreamConfig) -> "redis.Redis":
    return redis.Redis(
        host=config.host,
        port=config.port,
        db=config.db,
        password=config.password,
        decode_responses=True,
    )



def _encode(task_id: str, embedding: Sequence[float]) -> dict[str, str]:
    return {"payload": json.dumps({"task_id": task_id, "embedding": list(embedding)})}


def _decode(fields: dict[str, str]) -> tuple[str, list[float]]:
    if "payload" not in fields:
        raise MalformedMessageError(f"stream entry missing 'payload' field: {fields!r}")
    try:
        data = json.loads(fields["payload"])
        return data["task_id"], data["embedding"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise MalformedMessageError(f"could not decode payload {fields.get('payload')!r}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Publisher
# --------------------------------------------------------------------------- #

class RedisStreamPublisher:
    """Publishes classification tasks onto the stream. Used upstream of the
    worker (e.g. by the embedding-generation step, or by tests)."""

    def __init__(self, config: RedisStreamConfig | None = None, *, client: "redis.Redis | None" = None) -> None:
        self._config = config or RedisStreamConfig()
        self._client = client or _build_client(self._config)

    def publish(self, task_id: str, embedding: Sequence[float]) -> str:
        """Publish one task. Returns the Redis-assigned message id."""
        try:
            message_id = self._client.xadd(self._config.stream_name, _encode(task_id, embedding))
        except redis.exceptions.ConnectionError as exc:
            raise RedisConnectionUnavailable(f"could not reach Redis at {self._config.host}:{self._config.port}") from exc
        logger.info("published task_id=%s as message_id=%s", task_id, message_id)
        return message_id


# --------------------------------------------------------------------------- #
# Consumer
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StreamMessage:
    message_id: str
    task_id: str
    embedding: list[float]


class RedisStreamConsumer:

    def __init__(self, config: RedisStreamConfig | None = None, *, client: "redis.Redis | None" = None) -> None:
        self._config = config or RedisStreamConfig()
        self._client = client or _build_client(self._config)
        self._ensure_group()

    def _ensure_group(self) -> None:
        try:
            self._client.xgroup_create(
                name=self._config.stream_name,
                groupname=self._config.group_name,
                id="0",
                mkstream=True,
            )
            logger.info(
                "created consumer group %s on stream %s",
                self._config.group_name,
                self._config.stream_name,
            )
        except redis.exceptions.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise RedisStreamError(f"could not create consumer group: {exc}") from exc
        except redis.exceptions.ConnectionError as exc:
            raise RedisConnectionUnavailable(
                f"could not reach Redis at {self._config.host}:{self._config.port}"
            ) from exc

    def read(self) -> Iterator[StreamMessage]:

        try:
            response = self._client.xreadgroup(
                groupname=self._config.group_name,
                consumername=self._config.consumer_name,
                streams={self._config.stream_name: ">"},
                count=self._config.batch_size,
                block=self._config.block_ms,
            )
        except redis.exceptions.ConnectionError as exc:
            raise RedisConnectionUnavailable(
                f"could not reach Redis at {self._config.host}:{self._config.port}"
            ) from exc

        if not response:
            return

        for _stream_name, entries in response:
            for message_id, fields in entries:
                try:
                    task_id, embedding = _decode(fields)
                except MalformedMessageError:
                    logger.warning("dropping malformed message %s: %r", message_id, fields)
                    self.ack(message_id)
                    continue
                yield StreamMessage(message_id=message_id, task_id=task_id, embedding=embedding)

    def ack(self, message_id: str) -> None:
        self._client.xack(self._config.stream_name, self._config.group_name, message_id)

    def pending(self) -> list[dict[str, Any]]:

        summary = self._client.xpending(self._config.stream_name, self._config.group_name)
        return summary if isinstance(summary, list) else [summary]

    def close(self) -> None:
        self._client.close()
