/* eslint-disable indent */

const initialState = {
    notificationsystem: null,
}

function notifications(state = initialState, action) {
    switch (action.type) {
        case 'SET_SYS': {
            const { notificationsystem } = action.data
            return {
                ...state,
                notificationsystem
            }
        }
        default:
            return state
    }
}

export default notifications
