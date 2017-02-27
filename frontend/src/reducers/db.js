
import Horizon from '@horizon/client'

const onStatusChange = ({ type }) => {
    /* reload the page on error */
    if (type === 'error') {
        window.location = window.location.href // eslint-disable-line
    }
}

const horizon = Horizon({ secure: false })
horizon.status(onStatusChange)

const initialState = {
    horizon,
    trades: horizon('trades'),
    settings: horizon('settings'),
    notifications: horizon('notifications'),
}

function db(state = initialState, action) {
    switch (action.type) {
    default:
        return state
    }
}

export default db
