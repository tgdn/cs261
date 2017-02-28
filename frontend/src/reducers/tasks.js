/* eslint-disable indent */

const initialState = {
    tasks: [],
}

function tasks(state = initialState, action) {
    switch (action.type) {
        case 'UPDATE_TASKS': {
            return {
                ...state,
                tasks: action.data.tasks,
            }
        }
        default:
            return state
    }
}

export default tasks
