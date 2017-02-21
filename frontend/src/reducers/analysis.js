/* eslint-disable indent */

const initialState = {
    file: '',
    streamUrl: '',
    port: 80,
}

function analysis(state = initialState, action) {
    switch (action.type) {
        case 'UPDATE_FILE': {
            const { file } = action.data
            return {
                ...state,
                file,
            }
        }
        case 'UPDATE_STREAM': {
            const { streamUrl } = action.data
            return {
                ...state,
                streamUrl,
            }
        }
        case 'UPDATE_PORT': {
            const { port } = action.type
            return {
                ...state,
                port
            }
        }
        default:
            return state
    }
}

export default analysis
