
import Horizon from '@horizon/client'

const horizon = Horizon({ secure: false })

const initialState = {
    horizon,
    trades: horizon('trades'),
}

function db(state = initialState, action) {
    switch (action.type) {
    default:
        return state
    }
}

export default db
