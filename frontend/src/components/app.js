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
            .order('key')
            .watch()
            .subscribe((settings) => {
                this.props.updateSettings(settings)
            })
    }

    render() {
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
