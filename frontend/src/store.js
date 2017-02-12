/**
 * @file
 * Creates store, imports reducers, middlewares and redux devtools.
 * Add middlerwares in this file.
 *
 * Exports store as default.
 */

import {
    createStore,
    applyMiddleware,
    combineReducers,
    compose
} from 'redux'
import thunk from 'redux-thunk'

import { browserHistory } from 'react-router'
import { routerMiddleware, routerReducer } from 'react-router-redux'
import reducers from './reducers'

/*
    to push a route:
    import { push } from 'react-router-redux'
    store.dispatch(push('/some-route'))
*/

const historyMiddleware = routerMiddleware(browserHistory)

/* eslint-disable no-undef, no-underscore-dangle */
const composeEnhancers = window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ || compose

const store = createStore(
    combineReducers({
        ...reducers,
        routing: routerReducer,
    }),
    composeEnhancers(
        applyMiddleware(thunk),
        applyMiddleware(historyMiddleware),
    ),
)
/* eslint-enable */

export default store
