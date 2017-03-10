import React from 'react'
import { connect } from 'react-redux'
import { Link } from 'react-router'
import {
    Container,
    Grid,
    Icon,
    Header,
    Sidebar,
    Button,
} from 'semantic-ui-react'

import SymbolSidebar from '../symbolsidebar'

class HomePage extends React.Component {
    render() {
        const style = {
            height: `calc(100vh - ${this.props.headerHeight}px)`
        }
        const { alerts, alertCount } = this.props
        return (
            <div>
                <Sidebar.Pushable as='div' style={style}>
                    <SymbolSidebar symbols={this.props.symbols} />
                    <Sidebar.Pusher>
                        <Grid padded stackable>
                            <Grid.Column width={16}>
                                <Container fluid textAlign='center'>
                                    {alerts.length === 0 ? (
                                        <div>
                                            <Icon color='green' name='check circle' size='massive' class='pulsate' />
                                            <Header as='h2' color='green'>No issues</Header>
                                        </div>
                                    ) : (
                                        <div>
                                            <Icon color='red' name='remove circle' size='massive' class='pulsate' />
                                            <Header as='h2' color='red'>
                                                {alertCount === 1 ? (
                                                    'There is one issue to review'
                                                ) : (
                                                    `There are ${alertCount} issues to review`
                                                )}
                                            </Header>
                                            <Button as={Link} to='/flagged' secondary>
                                                Review now
                                            </Button>
                                        </div>
                                    )}
                                </Container>
                            </Grid.Column>
                        </Grid>
                    </Sidebar.Pusher>
                </Sidebar.Pushable>
            </div>
        )
    }
}

HomePage.propTypes = {
    headerHeight: React.PropTypes.number,
    alerts: React.PropTypes.array, // eslint-disable-line
    symbols: React.PropTypes.array, // eslint-disable-line
    alertCount: React.PropTypes.number,
}

HomePage.defaultProps = {
    headerHeight: 0,
    alerts: [],
    symbols: [],
    alertCount: 0,
}

export default connect(
    state => ({
        alerts: state.db.alerts,
        symbols: state.db.symbols,
        headerHeight: state.ui.headerHeight,
        alertCount: state.db.alertCount,
    }),
)(HomePage)
