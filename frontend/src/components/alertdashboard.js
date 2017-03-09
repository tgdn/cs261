import React, { PropTypes } from 'react'
import { connect } from 'react-redux'
import { browserHistory, Link } from 'react-router'

import {
    Container,
    Grid,
    Icon,
    Menu,
    Segment,
    Loader,
    Confirm,
    Label,
    Button
} from 'semantic-ui-react'

import SymbolChart from './symbolchart'

const severityToColor = (severity) => {
    switch (severity) {
    case 2: return 'yellow'
    case 3: return 'blue'
    case 1:
    default:
        return 'red'
    }
}

class AlertDashboard extends React.Component {
    constructor(props) {
        super(props)
        this.handleCancelAnomaly = this.handleCancelAnomaly.bind(this)
        this.state = {
            cancelModalOpened: false,
        }
    }

    handleOpenCancelAnomaly = () => this.setState({ cancelModalOpened: true })

    handleCancelAnomalyCancelBtn = () => this.setState({ cancelModalOpened: false })

    handleCancelAnomalyConfirmBtn = () => {
        this.setState({ cancelModalOpened: false })
        this.handleCancelAnomaly()
    }

    handleSuccessfulDelete() {
        this.props.notificationsystem.addNotification({
            level: 'info',
            title: 'Anomaly correctly discarded',
            message: 'The anomaly was successfully removed'
        })
    }

    cancelMultiple(id) {
        const horizon = this.props.horizon
        try {
            horizon('alerts').remove(id)
            this.handleSuccessfulDelete()
            browserHistory.push('/flagged')
        } catch (err) {
            console.log(err);
        }
    }

    cancelOne(alertid) {
        fetch('/api/alerts/delete', { // eslint-disable-line
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ alertid })
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
            if (res.success) {
                this.handleSuccessfulDelete()
                browserHistory.push('/flagged')
            }
        })
        .catch((err) => {
            console.error(err); // eslint-disable-line
            this.props.notificationsystem.addNotification({
                level: 'error',
                title: 'The anomaly couldn\'t be discarded',
                message: 'Something unexpected came up, try again soon'
            })
        })
    }

    handleCancelAnomaly() {
        const { alert } = this.props
        if (alert.trade_pk === -1) {
            this.cancelMultiple(alert.id)
        } else {
            this.cancelOne(alert.id)
        }
    }

    render() {
        const { alert, trades } = this.props
        return (
            <div>
                <Confirm
                    open={this.state.cancelModalOpened}
                    onCancel={this.handleCancelAnomalyCancelBtn}
                    onConfirm={this.handleCancelAnomalyConfirmBtn}
                    header='Are you sure you want to delete this anomaly?'
                    content='This action cannot be undone'
                    confirmButton='Yes'
                />
            <Menu borderless attached='top' inverted>
                    <Menu.Item header>
                        <Icon name='warning sign' />
                        {alert.description}
                        <Label color={severityToColor(alert.severity)}>{alert.severity}</Label>
                    </Menu.Item>
                    <Menu.Menu position='right'>
                        <Menu.Item as={Link} to={`/${alert.symbol}`}>
                            <Icon name='external' />
                            open {alert.symbol}
                        </Menu.Item>
                        <Menu.Item fitted>
                            <Button
                                inverted
                            >
                                Review
                            </Button>
                        </Menu.Item>
                        <Menu.Item>
                            <Button
                                inverted
                                color='orange'
                                onClick={this.handleOpenCancelAnomaly}
                            >
                                Not an anomaly?
                            </Button>
                        </Menu.Item>
                    </Menu.Menu>
                </Menu>
                <div>
                    <Segment inverted attached='bottom'>
                        {(trades.length === 0) ? (
                            <Loader active inverted size='large' inline='centered'>
                                Loading trades...
                            </Loader>
                        ) : (
                            <SymbolChart
                                symbol='random'
                                trades={trades}
                                flagAnomalies={false}
                                flagOne={alert.trade_pk}
                            />
                        )}
                    </Segment>
                </div>
            </div>
        )
    }
}

export default connect(
    state => ({
        notificationsystem: state.notifications.notificationsystem,
    })
)(AlertDashboard)
