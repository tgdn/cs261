import React from 'react'
import { connect } from 'react-redux'
import { Link } from 'react-router'
import {
    Container,
    Grid,
    Icon,
    Menu,
    Segment,
    Loader
} from 'semantic-ui-react'

import isEqual from 'lodash/isEqual'
import find from 'lodash/find'


class AlertPage extends React.Component {
    constructor(props) {
        super(props)
        this.state = {
            doesNotExist: false,
            alert: null,
            trades: [],
        }
    }

    componentWillMount() {
        this.getAlert(this.props.params.alertid)
    }

    componentWillReceiveProps(newProps) {
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
        let alert = find(this.props.alerts, obj => obj.id === alertid)
        if (alert === undefined) {
            this.props.horizon('alerts').find(alertid)
                .fetch()
                .subscribe((doc) => {
                    this.setState({ alert: doc })
                }, (err) => {
                    this.setState({ doesNotExist: true })
                })
        } else {
            this.setState({ alert })
        }
    }

    loadTrades(tradeid) {
        console.log('get trades');
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
                    <Container fluid textAlign='center'>
                        {this.state.alert === null ? (
                            <div>
                                <Segment inverted attached='bottom'>
                                    <Loader active inverted size='large' inline='centered'>
                                        Getting alert information...
                                    </Loader>
                                </Segment>
                            </div>
                        ) : (
                            <Menu size='huge' borderless attached='top' inverted>
                                <Menu.Item header>
                                    <Icon name='warning sign' />
                                    {alert.description}
                                </Menu.Item>
                                <Menu.Menu position='right'>
                                </Menu.Menu>
                            </Menu>
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
