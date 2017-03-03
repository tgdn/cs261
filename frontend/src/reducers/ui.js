
const initialState = {
    headerHeight: 0,
}

function ui(state = initialState, action) {
    switch (action.type) {
    case 'UI_SET_HEADER_HEIGHT': {
        return {
            ...state,
            headerHeight: action.data.headerHeight
        }
    }
    default:
        return state
    }
}

export default ui
