/* eslint-disable react/sort-comp */

import React from 'react'
import { connect } from 'react-redux'

import { Loader, Dimmer } from 'semantic-ui-react'
import Notification from './notification'
import HeaderNav from './header'

class App extends React.Component {
    state = {
        loaded: false,
    }

    componentDidMount() {
        /* eslint-disable no-undef */
        this.container = document.getElementById('app')
        this.header = document.getElementById('mainmenu')
        /* eslint-enable */

        this.subscribeNotifications()
        this.subscribeSettings()
        this.subscribeTasks()
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
                        message: notif.message,
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

        if (this.props.largetext) {
            App.addClass(document.documentElement, 'largetext-setting')
        } else {
            App.removeClass(document.documentElement, 'largetext-setting')
        }
        /* eslint-enable */
        const headerHeight = this.header.offsetHeight
        this.container.style.paddingTop = `${headerHeight}px`
    }

    render() {
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
                {this.props.children}
            </div>
        )
    }
}

export default connect(
    state => ({
        notifications: state.db.notifications,
        settings: state.db.settings,
        tasks: state.db.tasks,
        notificationsystem: state.notifications.notificationsystem,
        inverse: state.settings.inverse,
        largetext: state.settings.largetext
    }),
    dispatch => ({
        updateSettings: (settings) => {
            dispatch({
                type: 'UPDATE_SETTINGS',
                data: {
                    settings
                }
            })
        },
        updateTasks: (tasks) => {
            dispatch({
                type: 'UPDATE_TASKS',
                data: {
                    tasks
                }
            })
        }
    })
)(App)
