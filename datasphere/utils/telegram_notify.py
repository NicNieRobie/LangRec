import requests

from utils.auth import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DATASPHERE_COM, DATASPHERE_PROJ
from loguru import logger


class NotificationType:
    LAUNCH = 0
    ERROR_LAUNCH = 1
    ERROR_RUN = 2
    SUCCESS = 3


LAUNCH_MESSAGE_TEMPLATE = """
*Launched job* `{job_id}`

Task ID: `{task_id}`
Link: https://datasphere.yandex.cloud/communities/{datasphere_com}/projects/{datasphere_proj}/job/{job_id}
"""

ERROR_LAUNCH_MESSAGE_TEMPLATE = """
⚠️*Failed to run job for task* `{task_id}`

Error: **{error}**
"""

ERROR_RUN_MESSAGE_TEMPLATE = """
⚠️*Job* `{job_id}` *failed to run*

Task ID: `{task_id}`
Link: https://datasphere.yandex.cloud/communities/{datasphere_com}/projects/{datasphere_proj}/job/{job_id}
"""

SUCCESS_MESSAGE_TEMPLATE = """
*Job* `{job_id}` *finished successfully*

Task ID: `{task_id}`
Link: https://datasphere.yandex.cloud/communities/{datasphere_com}/projects/{datasphere_proj}/job/{job_id}
"""


def send_telegram_message(notification_type: NotificationType, data):
    if notification_type == NotificationType.LAUNCH:
        message = LAUNCH_MESSAGE_TEMPLATE.format(
            job_id=data.get('job_id'),
            task_id=data.get('task_id'),
            datasphere_com=DATASPHERE_COM,
            datasphere_proj=DATASPHERE_PROJ
        )
    elif notification_type == NotificationType.ERROR_LAUNCH:
        message = ERROR_LAUNCH_MESSAGE_TEMPLATE.format(
            task_id=data.get('task_id'),
            error=data.get('error'),
            datasphere_com=DATASPHERE_COM,
            datasphere_proj=DATASPHERE_PROJ
        )
    elif notification_type == NotificationType.ERROR_RUN:
        message = ERROR_RUN_MESSAGE_TEMPLATE.format(
            job_id=data.get('job_id'),
            task_id=data.get('task_id'),
            datasphere_com=DATASPHERE_COM,
            datasphere_proj=DATASPHERE_PROJ
        )
    else:
        message = SUCCESS_MESSAGE_TEMPLATE.format(
            job_id=data.get('job_id'),
            task_id=data.get('task_id'),
            datasphere_com=DATASPHERE_COM,
            datasphere_proj=DATASPHERE_PROJ
        )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, data=data)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to send Telegram message: {e}")
