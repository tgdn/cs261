/* eslint-disable react/sort-comp, react/forbid-prop-types */

import React, { PropTypes } from 'react'
import { connect } from 'react-redux'

import { Loader, Dimmer } from 'semantic-ui-react'
import Notification from './notification'
import HeaderNav from './header'
import NoDataComponent from './pages/nodata'

class App extends React.Component {
    state = {
        loaded: false,
        symbolsIntervalID: null,
    }

    constructor(props) {
        super(props)
        this.getSymbols = this.getSymbols.bind(this)
        this.subscribeAlertCount = this.subscribeAlertCount.bind(this)
    }

    componentDidMount() {
        /* eslint-disable no-undef */
        this.container = document.getElementById('app')
        this.header = document.getElementById('mainmenu')
        /* eslint-enable */

        /* realtime stuff */
        this.subscribeNotifications()
        this.subscribeSettings()
        this.subscribeAlerts()
        this.subscribeTasks()
        this.subscribeSymbols()
        this.subscribeAlertCount()
        setInterval(this.subscribeAlertCount, 8000)
    }

    subscribeNotifications() {
        this.props.notifications
            .order('datetime', 'descending')
            .watch({ rawChanges: true })
            .subscribe((notifChange) => {
                if (notifChange.type === 'add') {
                    const notif = notifChange.new_val
                    this.props.notificationsystem.addNotification({
                        allowHTML: true,
                        level: notif.level,
                        message: notif.message || null,
                        title: notif.title || null,
                    })
                }
            })
    }

    subscribeSettings() {
        this.props.settings
            .order('id')
            .watch()
            .subscribe((settings) => {
                this.setState({ loaded: true, })
                this.props.updateSettings(settings)
            })
    }

    subscribeTasks() {
        this.props.tasks
            .order('created_at')
            .findAll({ terminated: false })
            .watch()
            .subscribe((tasks) => {
                this.props.updateTasks(tasks)
            })
    }

    subscribeAlerts() {
        /* limit to 20 otherwise client will stall */
        this.props.horizon('alerts')
            .order('severity', 'ascending')
            .findAll({ reviewed: false, })
            .limit(20)
            .watch()
            .subscribe((alerts) => {
                this.props.updateAlerts(alerts)
            })
    }

    subscribeAlertCount() {
        fetch('/api/alertcount', { // eslint-disable-line
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then((res) => {
            if (res.status >= 200 && res.status < 300) {
                return res.json()
            }
            const err = new Error(res.statusText)
            err.response = res
            throw err
        })
        .then((res) => {
            this.props.setAlertCount(res.count)
        })
        .catch((err) => {
            console.error(err)
        })
    }

    subscribeSymbols() {
        /* get symbols every 5 seconds until we got all symbols.
         * There are 101 symbols, once reached, clear interval
        **/
        this.getSymbols() // load initial symbols
        if (this.state.symbolsIntervalID === null) {
            const symbolsIntervalID = setInterval(this.getSymbols, 5000)
            this.setState({ symbolsIntervalID })
        }
    }

    getSymbols() {
        /* check if we reached the maximum amount of symbols and return */
        if (this.props.symbols.length === 101 && this.state.symbolsIntervalID !== null) {
            clearInterval(this.state.symbolsIntervalID)
            return
        }

        fetch('/api/symbols') // eslint-disable-line
        .then((res) => {
            if (res.status >= 200 && res.status < 300) {
                return res.json()
            }
            const err = new Error(res.statusText)
            err.response = res
            throw err
        })
        .then((res) => {
            if (res.success && res.symbols) {
                this.props.updateSymbols(res.symbols)
            }
        })
        .catch((err) => {
            console.error(err)
        })
    }

    static addClass(el, className) {
        if (el.classList) {
            el.classList.add(className)
        } else {
            el.className += ` ${className}` // eslint-disable-line
        }
    }

    static removeClass(el, className) {
        if (el.classList) {
            el.classList.remove(className);
        } else {
            /* eslint-disable */
            el.className = el.className.replace(new RegExp('(^|\\b)' + className.split(' ').join('|') + '(\\b|$)', 'gi'), ' ');
            /* eslint-enable */
        }
    }

    componentDidUpdate() {
        /* eslint-disable */
        /* toggle invert colours */
        if (this.props.inverse) {
            App.addClass(document.documentElement, 'inverse-setting')
        } else {
            App.removeClass(document.documentElement, 'inverse-setting')
        }

        /* toggle large text */
        if (this.props.largetext) {
            App.addClass(document.documentElement, 'largetext-setting')
        } else {
            App.removeClass(document.documentElement, 'largetext-setting')
        }

        /* update header height */
        const headerHeight = this.header.offsetHeight
        this.props.setHeaderHeight(headerHeight) // update state
        this.container.style.paddingTop = `${headerHeight}px`
        /* eslint-enable */
    }

    render() {
        const nodata = this.props.symbols.length === 0
        return (
            <div>
                <Dimmer
                    active={!this.state.loaded}
                    page
                >
                    <Loader size='large'>Loading settings</Loader>
                </Dimmer>
                <Notification />
                <HeaderNav />
                {(this.state.loaded && nodata) && (
                    <NoDataComponent />
                )}
                {(this.state.loaded && !nodata) && (
                    this.props.children
                )}
            </div>
        )
    }
}

App.propTypes = {
    horizon: PropTypes.any,
    children: PropTypes.node,
    notifications: PropTypes.object,
    settings: PropTypes.object,
    tasks: PropTypes.object,
    symbols: PropTypes.array,
    notificationsystem: PropTypes.any,
    inverse: PropTypes.bool,
    largetext: PropTypes.bool,
    updateSettings: PropTypes.func,
    updateTasks: PropTypes.func,
    updateSymbols: PropTypes.func,
    setAlertCount: PropTypes.func,
}

App.defaultProps = {
    horizon: null,
    children: null,
    notifications: null,
    settings: null,
    tasks: null,
    symbols: [],
    notificationsystem: null,
    inverse: false,
    largetext: false,
    updateSettings: () => {},
    updateTasks: () => {},
    updateSymbols: () => {},
    setAlertCount: () => {},
}

export default connect(
    state => ({
        horizon: state.db.horizon,
        notifications: state.db.notifications,
        settings: state.db.settings,
        tasks: state.db.tasks,
        symbols: state.db.symbols,
        notificationsystem: state.notifications.notificationsystem,
        inverse: state.settings.inverse,
        largetext: state.settings.largetext,
    }),
    dispatch => ({
        setHeaderHeight: (headerHeight) => {
            dispatch({
                type: 'UI_SET_HEADER_HEIGHT',
                data: { headerHeight }
            })
        },
        updateSettings: (settings) => {
            dispatch({
                type: 'UPDATE_SETTINGS',
                data: { settings }
            })
        },
        updateAlerts: (alerts) => {
            dispatch({
                type: 'UPDATE_ALERTS',
                data: { alerts }
            })
        },
        updateTasks: (tasks) => {
            dispatch({
                type: 'UPDATE_TASKS',
                data: { tasks }
            })
        },
        updateSymbols: (symbols) => {
            dispatch({
                type: 'UPDATE_SYMBOLS',
                data: { symbols }
            })
        },
        setAlertCount: (count) => {
            dispatch({
                type: 'SET_ALERT_COUNT',
                data: { count }
            })
        },
    })
)(App)
