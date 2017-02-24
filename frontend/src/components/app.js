import React from 'react'
import { connect } from 'react-redux'

import Notification from './notification'
import HeaderNav from './header'

class App extends React.Component {
    componentDidMount() {
        this.initialSubscribe = true;
        this.subscribeNotifications()
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
    }),
)(App)
