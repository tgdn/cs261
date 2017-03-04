import React from 'react'
import { connect } from 'react-redux'
import { Link } from 'react-router'
import {
    Container,
    Grid,
    Header,
    Menu,
    Icon
} from 'semantic-ui-react'

import groupBy from 'lodash/groupBy'
import map from 'lodash/map'

class SymbolPage extends React.Component {
    constructor(props) {
        super(props)
        this.state = {
            loading: true,
            doesNotExist: false,
            symbols: this.props.symbols || []
        }
    }

    componentWillReceiveProps(newProps) {
        console.log(newProps.symbols);
        this.setState({
            symbols: newProps.symbols || []
        }, this.checkSymbol)
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
                            <Menu size='huge'>
                                <Menu.Item as={Link} to='/'>
                                    <Icon name='arrow left' />
                                    Return
                                </Menu.Item>
                            </Menu>
                            <Header as='h2'>{this.props.params.symbol}</Header>
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
