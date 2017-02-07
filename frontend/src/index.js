
import React from 'react'
import ReactDOM from 'react-dom'
import { Provider } from 'react-redux'
// import { syncHistoryWithStore } from 'react-router-redux'
// import {
//     applyRouterMiddleware,
//     Router,
//     browserHistory
// } from 'react-router'

import store from './store'
import App from './components/app'

// const history = syncHistoryWithStore(browserHistory, store);

ReactDOM.render(
    <Provider store={store}>
        <App />
    </Provider>,
    document.getElementById('app')
)
