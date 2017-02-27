/* eslint-disable indent */

import forEach from 'lodash/forEach'

const initialState = {
    mode: 'static',
    inverse: false,
    largetext: false,
}

function settings(state = initialState, action) {
    switch (action.type) {
        case 'SET_SETTING': {
            return {
                ...state,
                [action.data.id]: action.data.value,
            }
        }
        case 'UPDATE_SETTINGS': {
            const prefs = action.data.settings
            const newState = {}
            forEach(prefs, (setting) => {
                newState[setting.id] = setting.value
            })
            return {
                ...state,
                ...newState,
            }
        }
        default:
            return state
    }
}

export default settings
