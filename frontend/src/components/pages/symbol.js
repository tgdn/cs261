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
        const symbol = this.props.params.symbol
        const symbols = this.props.symbols
        for (let i = 0; i < symbols.length; i++) { // eslint-disable-line
            console.log(symbols[i].name);
            console.log(symbol);
            if (symbols[i].name === symbol) {
                return
            }
        }
        this.doesNotExist = true
    }

    renderDoesNotExist() {
        return (
            <Grid padded stackable>
                <Grid.Column width={16}>
                    <Container fluid>
                        <Header as='h2'>Symbol does not exist</Header>
                    </Container>
                </Grid.Column>
            </Grid>
        )
    }

    render() {
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
