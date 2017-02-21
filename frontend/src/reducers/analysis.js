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
        case 'SET_FILE': {
            const { file } = action.data
            // TODO: do request here
            return {
                ...state,
                file,
                streamUrl: '',
            }
        }
        case 'SET_STREAM': {
            const { streamUrl, port = 80 } = action.data
            // TODO: do request
            return {
                ...state,
                file: '',
                streamUrl,
                port,
            }
        }
        default:
            return state
    }
}

export default analysis
