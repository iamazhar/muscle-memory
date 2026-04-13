from __future__ import annotations

from muscle_memory.models import BackgroundJob, JobKind, JobStatus


def test_store_can_round_trip_background_job(tmp_db) -> None:
    job = BackgroundJob(kind=JobKind.EXTRACT, payload={"episode_id": "ep-1"})

    tmp_db.add_job(job)

    loaded = tmp_db.get_job(job.id)
    assert loaded is not None
    assert loaded.kind is JobKind.EXTRACT
    assert loaded.status is JobStatus.PENDING
    assert loaded.payload == {"episode_id": "ep-1"}


def test_store_lists_jobs_newest_first_and_filters_by_status(tmp_db) -> None:
    pending = BackgroundJob(kind=JobKind.EXTRACT, payload={"episode_id": "ep-1"})
    failed = BackgroundJob(kind=JobKind.REFINE, payload={}, status=JobStatus.FAILED, error="boom")
    tmp_db.add_job(pending)
    tmp_db.add_job(failed)

    failed_jobs = tmp_db.list_jobs(status=JobStatus.FAILED)

    assert len(failed_jobs) == 1
    assert failed_jobs[0].id == failed.id
    assert failed_jobs[0].error == "boom"


def test_store_updates_job_status_attempts_and_error(tmp_db) -> None:
    job = BackgroundJob(kind=JobKind.EXTRACT, payload={"episode_id": "ep-1"})
    tmp_db.add_job(job)

    tmp_db.update_job_status(job.id, status=JobStatus.RUNNING, attempts=1)
    tmp_db.update_job_status(job.id, status=JobStatus.FAILED, error="subprocess failed")

    loaded = tmp_db.get_job(job.id)
    assert loaded is not None
    assert loaded.status is JobStatus.FAILED
    assert loaded.attempts == 1
    assert loaded.error == "subprocess failed"
