from concurrent.futures._base import PENDING, RUNNING, FINISHED

from jsonschema._keywords import const

const()
APP_NAME = 'Simple Task Manager';
const()
API_BASE_URL = '/api/v1';
const()
TASK_STATUSES = {
    PENDING: 'Pending',
    RUNNING: 'In Progress',
    FINISHED: 'Completed'
};