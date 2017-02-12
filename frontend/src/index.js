
import React from 'react'
import ReactDOM from 'react-dom'
import { Provider } from 'react-redux'
import { syncHistoryWithStore } from 'react-router-redux'
import {
    Router,
    browserHistory
} from 'react-router'

import store from './store'
import routes from './routes'
// import App from './components/app'

const history = syncHistoryWithStore(browserHistory, store);

/* eslint-disable no-undef */
ReactDOM.render(
    <Provider store={store}>
        <Router history={history} routes={routes} />
    </Provider>,
    document.getElementById('app')
)
/* eslint-enable */
