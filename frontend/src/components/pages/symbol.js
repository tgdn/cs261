import React from 'react'
import { connect } from 'react-redux'
import { Link } from 'react-router'
import {
    Container,
    Grid,
    Header,
    Menu,
    Icon,
    Loader
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
        }
    }

    componentDidMount() {
        this.getTrades()
    }

    componentWillReceiveProps(newProps) {
        this.setState({
            symbols: newProps.symbols || []
        }, this.checkSymbol)
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
                this.setState({
                    loadingError: false,
                    loading: false,
                    trades: res.trades
                })
            }
        })
        .catch((err) => {
            this.setState({
                loadingError: true,
                loading: false,
            })
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
        return (
            <div>
                <Grid padded stackable>
                    <Grid.Column width={16}>
                        <Container fluid>
                            <Menu size='huge' borderless attached='top' inverted>
                                <Menu.Item as={Link} to='/'>
                                    <Icon name='arrow left' />
                                    Return
                                </Menu.Item>
                                <Menu.Item header>
                                    {this.props.params.symbol}
                                </Menu.Item>
                            </Menu>
                            {this.state.loading ? (
                                <Loader size='large' active inline='centered'>
                                    Getting latest stock data
                                </Loader>
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
    }),
)(SymbolPage)
