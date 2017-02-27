import React from 'react'
import { connect } from 'react-redux'
import { Link } from 'react-router'
import {
    Container,
    Grid,
    Icon,
    Header,
    Menu,
    Statistic,
    Dimmer,
    Loader
} from 'semantic-ui-react'

import groupBy from 'lodash/groupBy'
import map from 'lodash/map'

import io from 'socket.io-client'

class HomePage extends React.Component {
    constructor(props) {
        super(props)
        this.state = {
            loaded: true,
            trades: [],
        }
    }

    componentDidMount() {
        // this.subscribe()
        // this.subscribeIo()
    }

    subscribe() {
        this.props.trades
            .order('time')
            .watch()
            .subscribe((trades) => {
                this.setState({
                    loaded: true,
                    trades
                })
            })
    }

    subscribeIo() {
        this.s = io(`ws://${window.location.host}`) // eslint-disable-line no-undef
        this.s.on('connect', () => { console.log('connection open') })

        // s.on('disconnect', () => { console.log('disconnect') })
        // s.on('tradechange', (data) => { console.log(data) })
    }

    render() {
        return (
            <div>
                <Grid padded stackable>
                    <Grid.Column width={4}>
                        <Menu vertical fluid text>
                            {/* map(groupedTrades, (trades, symbol) => (
                                <Menu.Item key={symbol}>
                                    <Menu.Header>{symbol}</Menu.Header>
                                    <Menu.Menu>
                                        <Menu.Item name={`${trades.length} trades`} />
                                    </Menu.Menu>
                                </Menu.Item>
                            ))*/}
                        </Menu>
                    </Grid.Column>
                    <Grid.Column width={12}>
                        <Container textAlign='center'>
                            <Icon color='green' name='check circle' size='massive' />
                            <Header as='h2' color='green'>No issues</Header>
                        </Container>
                    </Grid.Column>
                </Grid>
            </div>
        )
    }
}

HomePage.propTypes = {
}

HomePage.defaultProps = {
}

export default connect(
    state => ({
    }),
)(HomePage)
