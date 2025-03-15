import asyncio
import concurrent.futures
import math
import multiprocessing
import multiprocessing.queues
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

from loguru import logger
from pydantic import BaseModel

from guidellm.config import settings
from guidellm.scheduler.strategy import (
    SchedulingStrategy,
    SynchronousStrategy,
    ThroughputStrategy,
)

__all__ = [
    "Scheduler",
    "SchedulerResult",
    "SchedulerRunInfo",
    "SchedulerRequestInfo",
    "RequestsWorker",
]


class RequestsWorker(ABC):
    """
    An abstract base class for a worker that processes requests.
    This class defines the interface for a worker that can resolve requests
    asynchronously or synchronously within the Scheduler class.
    Subclasses must implement the `resolve` method,
    which takes a request directly given from the load generator,
    along with the desired start_time for the request and a timeout_time.
    The `resolve` method should return the response from the backend.
    """

    @abstractmethod
    async def resolve(
        self,
        request: Any,
        timeout_time: float,
    ) -> Any:
        """
        An abstract method that must be implemented by subclasses.
        This method should handle the resolution of a request through asyncio,
        including any necessary backend processing and response handling.

        :param request: The request to be resolved generated by the load generator.
        :param timeout_time: The timeout time for the request, if there is no timeout
            given, then this will be math.inf.
        :return: The response from the worker.
        """
        ...


class SchedulerRunInfo(BaseModel):
    """
    Information about the current run of the scheduler.
    This class holds metadata about the scheduling run,
    including the start and end times, the number of processes,
    and the scheduling strategy used.
    It also tracks the number of requests created, queued, pending,
    and completed during the run.

    :param start_time: The start time of the scheduling run.
    :param end_time: The end time of the scheduling run;
        if None, then this will be math.inf.
    :param end_number: The maximum number of requests to be processed;
        if None, then this will be math.inf.
    :param processes: The number of processes used in the scheduling run.
    :param strategy: The scheduling strategy used in the run.
        This should be an instance of SchedulingStrategy.
    :param created_requests: The number of requests created during the run.
    :param queued_requests: The number of requests queued during the run.
    :param scheduled_requests: The number of requests scheduled during the run.
        (requests pending being sent to the worker but recieved by a process)
    :param processing_requests: The number of requests actively being run.
    :param completed_requests: The number of requests completed during the run.
    """

    start_time: float
    end_time: float
    end_number: float
    processes: int
    strategy: SchedulingStrategy

    created_requests: int = 0
    queued_requests: int = 0
    scheduled_requests: int = 0
    processing_requests: int = 0
    completed_requests: int = 0


class SchedulerRequestInfo(BaseModel):
    """
    Information about a specific request run through the scheduler.
    This class holds metadata about the request, including
    the targeted start time, queued time, start time, end time,
    and the process ID that handled the request.

    :param targeted_start_time: The targeted start time for the request (time.time()).
    :param queued_time: The time the request was queued (time.time()).
    :param scheduled_time: The time the request was scheduled (time.time())
        (any sleep time before the request was sent to the worker).
    :param worker_start: The time the worker started processing request (time.time()).
    :param worker_end: The time the worker finished processing request. (time.time()).
    :param process_id: The ID of the underlying process that handled the request.
    """

    targeted_start_time: float = -1
    queued_time: float = -1
    scheduled_time: float = -1
    worker_start: float = -1
    worker_end: float = -1
    process_id: int = -1


class SchedulerResult(BaseModel):
    """
    The yielded, iterative result for a scheduler run.
    These are triggered on the start and end of the run,
    as well as on the start and end of each request.
    Depending on the type, it will hold the request and response
    along with information and statistics about the request and general run.

    :param type_: The type of the result, which can be one of:
        - "run_start": Indicates the start of the run.
        - "run_complete": Indicates the completion of the run (teardown happens after).
        - "request_start": Indicates the start of a request.
        - "request_complete": Indicates the completion of a request.
    :param request: The request that was processed.
    :param response: The response from the worker for the request.
    :param request_info: Information about the request, including
        the targeted start time, queued time, start time, end time,
        and the process ID that handled the request.
    :param run_info: Information about the current run of the scheduler,
        including the start and end times, the number of processes,
        and the scheduling strategy used.
        It also tracks the number of requests created, queued, pending,
        and completed during the run.
    """

    type_: Literal[
        "run_start",
        "run_complete",
        "request_scheduled",
        "request_start",
        "request_complete",
    ]
    request: Any
    response: Any
    request_info: Optional[SchedulerRequestInfo]
    run_info: SchedulerRunInfo


@dataclass
class _WorkerProcessRequest:
    request: Any
    start_time: float
    timeout_time: Optional[float]
    queued_time: float


@dataclass
class _WorkerProcessResponse:
    type_: Literal["request_scheduled", "request_start", "request_complete"]
    request: Any
    response: Any
    info: SchedulerRequestInfo


class Scheduler:
    """
    A class that handles the scheduling of requests to a worker.
    This class is responsible for managing the lifecycle of the requests,
    including their creation, queuing, and processing.
    It uses a multiprocessing approach to handle requests concurrently
    and efficiently, based on the specified scheduling strategy.
    The Scheduler class is designed to work with a RequestsWorker,
    which is an abstract base class that defines the interface for a worker
    that can resolve requests asynchronously or synchronously.
    The Scheduler class also supports different scheduling strategies,
    including synchronous, throughput, and concurrent strategies.

    :param worker: The worker that will process the requests.
        This should be an instance of RequestsWorker.
    :param request_loader: An iterable that generates requests.
        This can be a list, generator, or any other iterable.
        The requests will be processed by the worker.
    :param scheduling_strategy: The scheduling strategy to use.
        Specifies the times at which requests will be sent as well how many
        worker processes are used and if requests are scheduled sync or async.
        This can be one of the following:
        - "synchronous": Requests are sent synchronously.
        - "throughput": Requests are sent at the maximum rate possible.
        - An instance of SchedulingStrategy.
    :param max_number: The maximum number of requests to process.
        If None, then no limit is set and either the iterator must be exhaustible
        or the max_duration must be set.
    :param max_duration: The maximum duration for the scheduling run.
        If None, then no limit is set and either the iterator must be exhaustible
        or the max_number must be set.
    :param num_processes: The number of processes to use for the worker.
        If None, then the number of processes is set to the number of CPU cores
        minus one, or the max_worker_processes setting if it is lower.
        If the scheduling strategy is synchronous, then this is set to 1.
        If the scheduling strategy is concurrent, then this is set to the number
        of streams in the strategy.
    """

    def __init__(
        self,
        worker: RequestsWorker,
        request_loader: Iterable[Any],
        scheduling_strategy: Union[
            Literal["synchronous", "throughput"], SchedulingStrategy
        ] = "throughput",
        max_number: Optional[int] = None,
        max_duration: Optional[float] = None,
        num_processes: Optional[int] = None,
    ):
        if not isinstance(worker, RequestsWorker):
            raise ValueError(f"Invalid worker: {worker}")

        if not isinstance(request_loader, Iterable):
            raise ValueError(f"Invalid request_loader: {request_loader}")

        if scheduling_strategy == "synchronous":
            scheduling_strategy = SynchronousStrategy()
        elif scheduling_strategy == "throughput":
            scheduling_strategy = ThroughputStrategy()

        if not isinstance(scheduling_strategy, SchedulingStrategy):
            raise ValueError(f"Invalid scheduling strategy: {scheduling_strategy}")

        self._worker = worker
        self._request_loader = request_loader
        self._scheduling_strategy: SchedulingStrategy = scheduling_strategy
        self._max_number = max_number
        self._max_duration = max_duration
        self._num_processes = num_processes

    async def run(self) -> AsyncGenerator[SchedulerResult, None]:
        """
        The main method that runs the scheduler.
        This method is a generator that yields SchedulerResult objects
        at the start and end of the run, as well as at the start and end
        of each request.
        It uses multiprocessing to handle requests concurrently
        and efficiently, based on the specified scheduling strategy.
        The method also handles the lifecycle of the requests,
        including their creation, queuing, and processing.
        The method is designed to be used as an asynchronous generator,
        allowing it to be used with asyncio and other asynchronous frameworks.

        :return: An asynchronous generator that yields SchedulerResult objects.
            Each SchedulerResult object contains information about the request,
            the response, and the run information.
        """

        with (
            multiprocessing.Manager() as manager,
            concurrent.futures.ProcessPoolExecutor() as executor,
        ):
            futures, requests_queue, responses_queue = await self._start_processes(
                manager, executor
            )
            run_info, requests_iter, times_iter = self._run_setup(futures)
            yield SchedulerResult(
                type_="run_start",
                request=None,
                response=None,
                request_info=None,
                run_info=run_info,
            )

            while True:
                if (
                    requests_iter is None
                    and run_info.completed_requests >= run_info.created_requests
                ):
                    # we've exhausted all requests we've wanted to run
                    # and yielded all responses
                    break

                if requests_iter is not None and not requests_queue.full():
                    # we have space on the queue, try to add more requests
                    # if we've reached the limit number/time or we've exhausted requests
                    # then set requests_iter to None to stop adding more
                    try:
                        request_time = next(times_iter)
                        if (run_info.queued_requests >= run_info.end_number) or (
                            request_time >= run_info.end_time
                        ):
                            raise StopIteration

                        request = next(requests_iter)
                        requests_queue.put(
                            _WorkerProcessRequest(
                                request=request,
                                start_time=request_time,
                                timeout_time=run_info.end_time,
                                queued_time=time.time(),
                            )
                        )
                        run_info.created_requests += 1
                        run_info.queued_requests += 1
                    except StopIteration:
                        requests_iter = None

                try:
                    process_response: _WorkerProcessResponse = (
                        responses_queue.get_nowait()
                    )

                    if process_response.type_ == "request_scheduled":
                        run_info.queued_requests -= 1
                        run_info.scheduled_requests += 1
                        yield SchedulerResult(
                            type_="request_scheduled",
                            request=process_response.request,
                            response=None,
                            request_info=process_response.info,
                            run_info=run_info,
                        )
                    elif process_response.type_ == "request_start":
                        run_info.scheduled_requests -= 1
                        run_info.processing_requests += 1
                        yield SchedulerResult(
                            type_="request_start",
                            request=process_response.request,
                            response=None,
                            request_info=process_response.info,
                            run_info=run_info,
                        )
                    elif process_response.type_ == "request_complete":
                        run_info.processing_requests -= 1
                        run_info.completed_requests += 1
                        yield SchedulerResult(
                            type_="request_complete",
                            request=process_response.request,
                            response=process_response.response,
                            request_info=process_response.info,
                            run_info=run_info,
                        )
                    else:
                        raise ValueError(
                            f"Invalid process response type: {process_response}"
                        )
                except multiprocessing.queues.Empty:
                    pass

                # yield control to the event loop
                await asyncio.sleep(settings.default_async_loop_sleep)

            yield SchedulerResult(
                type_="run_complete",
                request=None,
                response=None,
                request_info=None,
                run_info=run_info,
            )

            await self._stop_processes(futures, requests_queue)

    def _run_setup(
        self, processes: List[asyncio.Future]
    ) -> Tuple[SchedulerRunInfo, Iterator[Any], Iterator[float]]:
        requests_iter = iter(self._request_loader)
        start_time = time.time()
        times_iter = iter(self._scheduling_strategy.request_times())
        end_time = time.time() + (self._max_duration or math.inf)
        end_number = self._max_number or math.inf

        try:
            # update end number if the request loader is finite and less than max
            iter_length = len(self._request_loader)
            if 0 < iter_length < end_number:
                end_number = iter_length
        except TypeError:
            pass

        if end_number == math.inf and end_time is None:
            logger.warning(
                "No end number or end time set, "
                "scheduler will run indefinitely until the request loader is exhausted."
            )

        info = SchedulerRunInfo(
            start_time=start_time,
            end_time=end_time,
            end_number=end_number,
            processes=len(processes),
            strategy=self._scheduling_strategy,
        )

        return info, requests_iter, times_iter

    async def _start_processes(
        self,
        manager,
        executor: concurrent.futures.ProcessPoolExecutor,
    ) -> Tuple[
        List[asyncio.Future],
        multiprocessing.Queue,
        multiprocessing.Queue,
    ]:
        processing_mode = self._scheduling_strategy.processing_mode

        num_processes = self._scheduling_strategy.processes_limit
        if num_processes is None:
            cpu_cores = os.cpu_count() or 1
            num_processes = min(max(1, cpu_cores - 1), settings.max_worker_processes)

        num_processing_requests = self._scheduling_strategy.processing_requests_limit
        if num_processing_requests is None:
            num_processing_requests = settings.max_concurrency
        num_processing_requests_per_process = num_processing_requests // num_processes

        num_queued_requests = self._scheduling_strategy.queued_requests_limit
        if num_queued_requests is None:
            num_queued_requests = num_processing_requests + num_processes

        requests_queue = manager.Queue(maxsize=num_queued_requests)
        responses_queue = manager.Queue()

        futures = []
        loop = asyncio.get_event_loop()
        for process_id in range(num_processes):
            if processing_mode == "sync":
                futures.append(
                    loop.run_in_executor(
                        executor,
                        self._worker_process_sync,
                        requests_queue,
                        responses_queue,
                        process_id,
                    )
                )
            elif processing_mode == "async":
                futures.append(
                    loop.run_in_executor(
                        executor,
                        self._worker_process_async,
                        requests_queue,
                        responses_queue,
                        num_processing_requests_per_process,
                        process_id,
                    )
                )
            else:
                raise ValueError(
                    f"Invalid processing mode: {processing_mode} "
                    f"for strategy: {self._scheduling_strategy}"
                )

        await asyncio.sleep(0.1)  # give time for processes to start

        return futures, requests_queue, responses_queue

    async def _stop_processes(
        self,
        futures: List[asyncio.Future],
        requests_queue: multiprocessing.Queue,
    ):
        for _ in futures:
            requests_queue.put(None)

        await asyncio.gather(*futures)

    def _worker_process_sync(
        self,
        requests_queue: multiprocessing.Queue,
        results_queue: multiprocessing.Queue,
        process_id: int,
    ):
        async def _process_runner():
            while True:
                try:
                    process_request: Optional[_WorkerProcessRequest] = (
                        requests_queue.get_nowait()
                    )
                except multiprocessing.queues.Empty:
                    # yield control to the event loop
                    await asyncio.sleep(settings.default_async_loop_sleep)
                    continue

                if process_request is None:  # stop signal
                    break

                await self._worker_schedule_request(
                    worker=self._worker,
                    request=process_request.request,
                    queued_time=process_request.queued_time,
                    start_time=process_request.start_time,
                    timeout_time=process_request.timeout_time,
                    results_queue=results_queue,
                    process_id=process_id,
                )
                # yield control to event loop
                await asyncio.sleep(settings.default_async_loop_sleep)

        try:
            asyncio.run(_process_runner())
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"Error in worker process {process_id}: {exc}",
                exc_info=True,
                stack_info=True,
            )

    def _worker_process_async(
        self,
        requests_queue: multiprocessing.Queue,
        results_queue: multiprocessing.Queue,
        max_concurrency: Optional[int],
        process_id: int,
    ):
        async def _process_runner():
            pending = asyncio.Semaphore(max_concurrency) if max_concurrency else None

            while True:
                try:
                    process_request: Optional[_WorkerProcessRequest] = (
                        requests_queue.get_nowait()
                    )
                except multiprocessing.queues.Empty:
                    # yield control to event loop
                    await asyncio.sleep(settings.default_async_loop_sleep)
                    continue

                if process_request is None:  # stop signal
                    break

                if pending:
                    await pending.acquire()

                def _task_done(_: asyncio.Task):
                    nonlocal pending
                    if pending:
                        pending.release()

                task = asyncio.create_task(
                    self._worker_schedule_request(
                        worker=self._worker,
                        request=process_request.request,
                        queued_time=process_request.queued_time,
                        start_time=process_request.start_time,
                        timeout_time=process_request.timeout_time,
                        results_queue=results_queue,
                        process_id=process_id,
                    )
                )
                task.add_done_callback(_task_done)
                # yield control to event loop
                await asyncio.sleep(settings.default_async_loop_sleep)

        try:
            asyncio.run(_process_runner())
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"Error in worker process {process_id}: {exc}",
                exc_info=True,
                stack_info=True,
            )

    @staticmethod
    async def _worker_schedule_request(
        worker: RequestsWorker,
        request: Any,
        queued_time: float,
        start_time: float,
        timeout_time: float,
        results_queue: multiprocessing.Queue,
        process_id: int,
    ):
        info = SchedulerRequestInfo(
            targeted_start_time=start_time,
            queued_time=queued_time,
            scheduled_time=time.time(),
            worker_start=-1,
            worker_end=-1,
            process_id=process_id,
        )
        results_queue.put(
            _WorkerProcessResponse(
                type_="request_scheduled",
                request=request,
                response=None,
                info=info,
            )
        )

        if (wait_time := start_time - time.time()) > 0:
            await asyncio.sleep(wait_time)

        info.worker_start = time.time()
        results_queue.put(
            _WorkerProcessResponse(
                type_="request_start",
                request=request,
                response=None,
                info=info,
            )
        )

        response = await worker.resolve(request, timeout_time)

        info.worker_end = time.time()
        results_queue.put(
            _WorkerProcessResponse(
                type_="request_complete",
                request=request,
                response=response,
                info=info,
            )
        )
