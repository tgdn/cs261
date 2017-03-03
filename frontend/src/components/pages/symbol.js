import React from 'react'
import { connect } from 'react-redux'
import { Link } from 'react-router'
import {
    Container,
    Grid,
    Header,
    Menu,
} from 'semantic-ui-react'

import groupBy from 'lodash/groupBy'
import map from 'lodash/map'

class SymbolPage extends React.Component {
    render() {
        return (
            <Grid padded stackable>
                <Grid.Column width={16}>
                    <Container>
                        <Header as='h2'>{this.props.params.symbol}</Header>
                    </Container>
                </Grid.Column>
            </Grid>
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
