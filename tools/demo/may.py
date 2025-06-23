import time
import types
import uuid

import celery
import requests
from celery import Task
from celery.utils.log import get_task_logger
from kombu import Queue


class MayTask(Task):
    MAIN_NAME = "ProjectMay"
    acks_late = True
    autoretry_for = (Exception,)

    def __init__(
        self,
        may_url: str = "https://projectmay.davidz.cn/api",
        flow_id: str | None = None,
        task_id: str | None = None,
        task_name: str | None = None,
        dependencies: list[str] | None = None,
        **celery_kwargs,
    ):
        for k, v in celery_kwargs.items():
            setattr(self, k, v)

        if dependencies is None:
            dependencies = []

        self._get_task_info(may_url, flow_id, task_id, task_name, dependencies)
        self.app = celery.Celery("ProjectMay", broker=self._broker_url)
        self.app.conf.worker_hijack_root_logger = False
        self.app.conf.worker_redirect_stdouts = False
        self.name = f"{self.MAIN_NAME}.{self._task['id']}"
        self.app.conf.task_queues = (Queue(self.name, routing_key=self.name),)
        super().__init__()
        self.app.register_task(self)
        self.logger = get_task_logger(self.name)

    def _get_task_info(self, may_url, flow_id, task_id, task_name, dependencies):
        try:
            r = requests.post(
                f"{may_url}/new_task_worker/",
                json={
                    "flow_id": flow_id,
                    "task_id": task_id,
                    "task_name": task_name,
                    "dependencies": dependencies,
                },
                timeout=30,
            )
            if r.status_code != 201:
                raise Exception(f"Failed to get task info: {r.content}")
            r = r.json()
            self._task = r["task"]
            self._flow = r["task"]["flow"]
            self._broker_url = r["settings"]["CELERY_BROKER_URL"]

            ### HACK: jump server
            if "davidz.cn" in self._broker_url:
                self._broker_url = self._broker_url.replace("davidz.cn", "jump.davidz.cn")
            ### END HACK

        except requests.RequestException as e:
            raise Exception(f"Network error when getting task info: {e}")
        except KeyError as e:
            raise Exception(f"Missing key in response: {e}")

    def _get_parent_task_result_id(self):
        return self.request.headers.get("parent_task_result_id")

    def save_result(self, result: dict, success: bool = True):
        end_time = time.time()
        start_time = self.request.headers["start_time"]
        save_task_result_name = f"{self.MAIN_NAME}.save_task_result"
        self.app.send_task(
            save_task_result_name,
            kwargs=dict(task_id=self._task["id"], success=success, payload=result),
            queue=save_task_result_name,
            headers=dict(
                parent_task_result_id=self._get_parent_task_result_id(), start_time=start_time, end_time=end_time
            ),
        )

    def run_server(self, *cli_args):
        argv = [
            "worker",
            f"--hostname={self.name}@%h({uuid.uuid4()})",
            "--concurrency=1",
            "--prefetch-multiplier=1",
            "--pool=solo",
            "--loglevel=INFO",
        ]
        for arg in cli_args:
            argv.append(arg)
        self.app.worker_main(argv=argv)

    def before_start(self, task_id, args, kwargs):
        if not hasattr(self.request, "headers"):
            self.request.headers = {}
        self.request.headers["start_time"] = time.time()
        return super().before_start(task_id, args, kwargs)

    def run(self, payload: dict):
        raise NotImplementedError()


def may_task(
    may_url: str = "http://projectmay.davidz.cn/api",
    flow_id: str | None = None,
    task_id: str | None = None,
    task_name: str | None = None,
    dependencies: list[str] | None = None,
    **celery_kwargs,
):
    def decorator(func):
        task_instance = MayTask(
            may_url=may_url,
            flow_id=flow_id,
            task_id=task_id,
            task_name=task_name,
            dependencies=dependencies,
            **celery_kwargs,
        )

        task_instance.run = types.MethodType(func, task_instance)

        return task_instance

    return decorator
