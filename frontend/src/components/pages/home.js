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
        const tradeCount = parseInt(this.state.trades.length, 10)

        const items = [
          { label: 'Trades', value: tradeCount },
          { label: 'Issues', value: '0' },
          { label: 'Unresolved', value: '0' },
        ]

        const groupedTrades = groupBy(this.state.trades, 'symbol')

        return (
            <div>
                <Dimmer
                    active={!this.state.loaded}
                    page
                >
                    <Loader size='large'>Loading latest data</Loader>
                </Dimmer>
                <Grid padded stackable>
                    <Grid.Row color='black'>
                        <Grid.Column width={16}>
                            <Statistic.Group
                                inverted
                                widths='three'
                                items={items}
                                color='green'
                            />
                        </Grid.Column>
                    </Grid.Row>
                    <Grid.Column width={4}>
                        <Menu vertical>
                            {map(groupedTrades, (trades, symbol) => (
                                <Menu.Item key={symbol}>
                                    <Menu.Header>{symbol}</Menu.Header>
                                    <Menu.Menu>
                                        <Menu.Item name={`${trades.length} trades`} />
                                    </Menu.Menu>
                                </Menu.Item>
                            ))}
                        </Menu>
                    </Grid.Column>
                    <Grid.Column width={12}>
                        <Container textAlign='left'>
                            <Link to='/some/random/url'>Some random url</Link>
                        </Container>
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
    trades: React.PropTypes.object, // eslint-disable-line
}

HomePage.defaultProps = {
    trades: [],
}

export default connect(
    state => ({
        trades: state.db.trades,
    }),
)(HomePage)
