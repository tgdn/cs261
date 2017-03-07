/* eslint-disable react/sort-comp */

import React from 'react'
import { connect } from 'react-redux'
import { Link } from 'react-router'
import {
    Container,
    Grid,
    Segment,
    Loader
} from 'semantic-ui-react'

import AlertDashboard from '../alertdashboard'

import isEqual from 'lodash/isEqual'
import find from 'lodash/find'


class AlertPage extends React.Component {

    initialState = {
        doesNotExist: false,
        alert: null,
        trades: [],
    }

    constructor(props) {
        super(props)
        this.state = this.initialState
    }

    componentWillMount() {
        this.getAlert(this.props.params.alertid)
    }

    componentDidMount() {
        this.mounted = true
    }

    componentWillUnmount() {
        this.mounted = false
    }

    componentWillReceiveProps(newProps) {
        this.setState(this.initialState)
        if (newProps.params.alertid !== this.props.params.alertid) {
            this.getAlert(newProps.params.alertid)
        }
    }

    componentWillUpdate(newProps, newState) {
        // console.log('will update')
        /* check whether we're loading another alert */
        // const notNull = (newState.alert != null && this.state.alert != null)
        // if (newState.alert !== this.state.alert && newState.alert != null) {
        //     /* if we just loaded the alert */
        //     this.loadTrades(newState.alert.trade_pk)
        // }
    }

    getAlert(alertid) {
        const alert = find(this.props.alerts, obj => obj.id === alertid)
        if (alert === undefined) {
            this.props.horizon('alerts').find(alertid)
                .fetch()
                .subscribe((doc) => {
                    if (this.mounted) {
                        if (doc != null) {
                            this.setState(
                                { alert: doc },
                                () => this.loadTrades(doc.trade_pk)
                            )
                        } else {
                            this.setState({ doesNotExist: true })
                        }
                    }
                }, (err) => {
                    // TODO
                })
        } else {
            this.setState(
                { alert },
                () => this.loadTrades(alert.trade_pk)
            )
        }
    }

    loadTrades(tradeid) {
        fetch(`/api/trades/flagged/${tradeid}`) // eslint-disable-line
        .then((res) => {
            if (res.status >= 200 && res.status < 300) {
                return res.json()
            }
            const err = new Error(res.statusText)
            err.response = res
            throw err
        })
        .then((res) => {
            if (this.mounted) {
                this.setState({
                    trades: res.trades,
                })
            }
        })
        .catch((err) => {
            if (this.mounted) {
                // TODO
            }
        })
        // TODO fetch
    }

    render() {
        const { alert, trades } = this.state
        if (this.state.doesNotExist) {
            // TODO render does not exist
            return <h1>Does not exist</h1>
        }
        return (
            <Grid padded stackable>
                <Grid.Column width={16}>
                    <Container fluid className='chartContainer'>
                        {this.state.alert === null ? (
                            <div>
                                <Segment inverted attached='bottom'>
                                    <Loader active inverted size='large' inline='centered'>
                                        Getting alert information...
                                    </Loader>
                                </Segment>
                            </div>
                        ) : (
                            <AlertDashboard alert={alert} trades={trades} />
                        )}
                    </Container>
                </Grid.Column>
            </Grid>
        )
    }
}

/* eslint-disable react/forbid-prop-types */
AlertPage.propTypes = {
    horizon: React.PropTypes.any,
    alerts: React.PropTypes.array,
    params: React.PropTypes.object,
}

AlertPage.defaultProps = {
    horizon: null,
    alerts: [],
    params: {},
}
/* eslint-enable */

export default connect(
    state => ({
        horizon: state.db.horizon,
        alerts: state.db.alerts,
    }),
)(AlertPage)
