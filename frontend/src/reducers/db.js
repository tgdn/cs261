
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
    default:
        return state
    }
}

export default db
