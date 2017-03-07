/* eslint-disable react/sort-comp */

import React from 'react'
import { connect } from 'react-redux'
import { Link } from 'react-router'
import {
    Container,
    Grid,
    Header,
    Menu,
    Icon,
    Loader,
    Segment
} from 'semantic-ui-react'

import SymbolDashboard from '../symboldashboard'

import groupBy from 'lodash/groupBy'
import map from 'lodash/map'

class SymbolPage extends React.Component {
    constructor(props) {
        super(props)
        this.state = {
            loading: true,
            loadingError: false,
            doesNotExist: false,
            symbols: this.props.symbols || [],
            trades: [],
            pollIntervalID: null,
            liveTrades: true,
        }
        this.toggleLive = this.toggleLive.bind(this)
        this.getTrades = this.getTrades.bind(this)
    }

    componentWillMount() {
        /* trades are live as soon as we mount,
        check whether there arent any tasks,
        as in this case we can disable live trades. */
        if (this.props.tasks.length === 0 && this.state.liveTrades) {
            this.toggleLive()
        }
    }

    componentDidMount() {
        this.mounted = true
        this.subscribeTrades()
    }

    componentWillUnmount() {
        // stop polling once unmounted
        this.mounted = false
        if (this.state.pollIntervalID !== null) {
            clearInterval(this.state.pollIntervalID)
        }
    }

    componentWillReceiveProps(newProps) {
        const tasks = newProps.tasks
        /* disable live feed when there is no analysis -> and vice-versa */
        if (tasks.length === 0 && this.state.liveTrades) {
            this.toggleLive()
        } else if (tasks.length !== 0 && !this.state.liveTrades) {
            this.toggleLive()
        }

        this.setState({
            symbols: newProps.symbols || []
        }, this.checkSymbol)
    }

    subscribeTrades() {
        this.getTrades()
        if (this.state.pollIntervalID === null) {
            const pollIntervalID = setInterval(this.getTrades, 1000)
            this.setState({ pollIntervalID })
        }
    }

    toggleLive() {
        if (this.state.liveTrades && this.state.pollIntervalID) {
            clearInterval(this.state.pollIntervalID)
            this.setState({ pollIntervalID: null })
        } else if (!this.state.liveTrades) {
            this.subscribeTrades()
        }
        this.setState({ liveTrades: !this.state.liveTrades })
    }

    getTrades() {
        fetch(`/api/symbol/${this.props.params.symbol}`) // eslint-disable-line
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
                if (this.mounted) {
                    this.setState({
                        loadingError: false,
                        loading: false,
                        trades: res.trades
                    })
                }
            }
        })
        .catch((err) => {
            if (this.mounted) {
                this.setState({
                    loadingError: true,
                    loading: false,
                })
            }
            console.error(err);
        })
    }

    checkSymbol() {
        const symbol = this.props.params.symbol
        for (let i = 0; i < this.state.symbols.length; i++) { // eslint-disable-line
            if (this.state.symbols[i].name === symbol) {
                this.setState({ doesNotExist: false })
                return
            }
        }
        this.setState({ doesNotExist: true })
    }

    renderDoesNotExist() {
        return (
            <Grid padded stackable>
                <Grid.Column width={16}>
                    <Container fluid>
                        <Header as='h2'>
                            <strong>{this.props.params.symbol}</strong> does not exist
                        </Header>
                    </Container>
                </Grid.Column>
            </Grid>
        )
    }

    render() {
        if (this.state.doesNotExist) {
            return this.renderDoesNotExist()
        }
        const live = this.state.liveTrades
        return (
            <div>
                <Grid padded stackable>
                    <Grid.Column width={16}>
                        <Container fluid className='chartContainer'>
                            <Menu size='huge' borderless attached='top' inverted>
                                <Menu.Item as={Link} to='/'>
                                    <Icon name='arrow left' />
                                    Return
                                </Menu.Item>
                                <Menu.Item header>
                                    {this.props.params.symbol}
                                </Menu.Item>
                                <Menu.Menu position='right'>
                                    <Menu.Item onClick={this.toggleLive}>
                                        {live ? 'pause' : 'resume'}
                                    </Menu.Item>
                                    <Menu.Item>
                                        {live ? (
                                            <span>
                                                live <i class='indicator-green' />
                                            </span>
                                        ) : (
                                            <span>
                                                paused <i class='indicator-red' />
                                            </span>
                                        )}
                                    </Menu.Item>
                                </Menu.Menu>
                            </Menu>
                            {this.state.loading ? (
                                <div>
                                    <Segment inverted attached='bottom'>
                                        <Loader active inverted size='large' inline='centered'>
                                            Getting latest stock data
                                        </Loader>
                                    </Segment>
                                </div>
                            ) : (
                                <SymbolDashboard
                                    symbol={this.props.params.symbol}
                                    trades={this.state.trades}
                                    loadingError={this.state.loadingError}
                                />
                            )}
                        </Container>
                    </Grid.Column>
                </Grid>
            </div>
        )
    }
}

SymbolPage.propTypes = {
}

SymbolPage.defaultProps = {
}

export default connect(
    state => ({
        symbols: state.db.symbols,
        tasks: state.tasks.tasks,
    }),
)(SymbolPage)
