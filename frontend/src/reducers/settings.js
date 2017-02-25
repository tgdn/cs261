/* eslint-disable indent */

import forEach from 'lodash/forEach'

const initialState = {
    mode: 'static',
    inverse: false,
}

function settings(state = initialState, action) {
    switch (action.type) {
        case 'SET_SETTING': {
            return {
                ...state,
                [action.data.key]: action.data.value,
            }
        }
        case 'UPDATE_SETTINGS': {
            const prefs = action.data.settings
            const newState = {}
            forEach(prefs, (setting) => {
                newState[setting.key] = setting.value
            })
            return {
                ...state,
                ...newState,
            }
        }
        default:
            return initialState
    }
}

export default settings
