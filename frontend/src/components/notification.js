import React from 'react'
import { connect } from 'react-redux'
import NotificationSystem from 'react-notification-system'

class Notification extends React.Component {
    componentDidMount() {
        this.props.setSystem(this.notificationsystem)
    }

    render() {
        return (
            <NotificationSystem
                allowHTML={true}
                ref={(el) => { this.notificationsystem = el }}
            />
        )
    }
}

Notification.propTypes = {
    setSystem: React.PropTypes.func,
}

Notification.defaultProps = {
    setSystem: () => {}
}

export default connect(
    () => ({}),
    dispatch => ({
        setSystem: (notificationsystem) => {
            dispatch({
                type: 'SET_SYS',
                data: {
                    notificationsystem
                }
            })
        }
    })
)(Notification)
