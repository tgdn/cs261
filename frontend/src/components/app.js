/* eslint-disable react/sort-comp */

import React from 'react'
import { connect } from 'react-redux'

import Notification from './notification'
import HeaderNav from './header'

class App extends React.Component {
    componentDidMount() {
        this.initialSubscribe = true;
        this.subscribeNotifications()
        this.subscribeSettings()
    }

    subscribeNotifications() {
        this.props.notifications
            .order('datetime')
            .limit(10)
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
                this.props.updateSettings(settings)
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

    render() {
        /* toggle invert colours */
        if (this.props.inverse) {
            App.addClass(document.documentElement, 'inverse-setting') // eslint-disable-line no-undef
        } else {
            App.removeClass(document.documentElement, 'inverse-setting') // eslint-disable-line
        }

        return (
            <div>
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
        notificationsystem: state.notifications.notificationsystem,
        settings: state.db.settings,
        inverse: state.settings.inverse,
    }),
    dispatch => ({
        updateSettings: (settings) => {
            dispatch({
                type: 'UPDATE_SETTINGS',
                data: {
                    settings
                }
            })
        }
    })
)(App)
