
import Horizon from '@horizon/client'

const onStatusChange = ({ type }) => {
    /* eslint-disable no-undef */
    /* reload the page on error */
    if (type === 'error') {
        window.location = window.location.href // can fail
        window.location.reload() // if page didnt reload yet, hopefully its not post
    }
    /* eslint-enable */
}

const horizon = Horizon({ secure: false })
horizon.status(onStatusChange)

const initialState = {
    horizon,
    trades: horizon('trades'),
    settings: horizon('settings'),
    alerts: [],
    alertCount: 0,
    notifications: horizon('notifications'),
    tasks: horizon('tasks'),
    symbols: [],
}

function db(state = initialState, action) {
    switch (action.type) {
    case 'UPDATE_SYMBOLS': {
        return {
            ...state,
            symbols: action.data.symbols || [],
        }
    }
    case 'UPDATE_ALERTS': {
        return {
            ...state,
            alerts: action.data.alerts || [],
        }
    }
    case 'SET_ALERT_COUNT': {
        return {
            ...state,
            alertCount: action.data.count || 0
        }
    }
    case 'INCREMENT_ALERT_COUNT': {
        return {
            ...state,
            alertCount: state.alertCount + 1
        }
    }
    case 'DECREMENT_ALERT_COUNT': {
        return {
            ...state,
            alertCount: state.alertCount - 1
        }
    }
    default:
        return state
    }
}

export default db
