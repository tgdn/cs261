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

import AlertsSidebar from '../alertssidebar'

class FlaggedPage extends React.Component {
    renderNoChildren = () => (
        <Grid padded stackable>
            <Grid.Column width={16}>
                <Container fluid textAlign='center'>
                    <Icon name='arrow left' size='massive' color='grey' />
                    <Header as='h1' color='grey'>
                        Pick an alert on the left to start reviewing
                    </Header>
                </Container>
            </Grid.Column>
        </Grid>
    )
    render() {
        const style = {
            height: `calc(100vh - ${this.props.headerHeight}px)`
        }
        return (
            <div>
                <Sidebar.Pushable as='div' style={style}>
                    <AlertsSidebar
                        horizon={this.props.horizon}
                        alerts={this.props.alerts}
                        params={this.props.params}
                    />
                    <Sidebar.Pusher>
                        {this.props.children === null ? (
                            this.renderNoChildren()
                        ) : this.props.children}
                    </Sidebar.Pusher>
                </Sidebar.Pushable>
            </div>
        )
    }
}

FlaggedPage.propTypes = {
    children: React.PropTypes.node,
    headerHeight: React.PropTypes.number,
    horizon: React.PropTypes.any, // eslint-disable-line
    alerts: React.PropTypes.array, // eslint-disable-line
}

FlaggedPage.defaultProps = {
    children: null,
    headerHeight: 0,
    horizon: null,
    alerts: [],
}

export default connect(
    state => ({
        horizon: state.db.horizon,
        alerts: state.db.alerts,
        headerHeight: state.ui.headerHeight,
    }),
)(FlaggedPage)
