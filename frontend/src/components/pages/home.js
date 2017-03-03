import React from 'react'
import { connect } from 'react-redux'
import {
    Container,
    Grid,
    Icon,
    Header,
    Sidebar,
} from 'semantic-ui-react'

import SymbolSidebar from '../symbolsidebar'

// import io from 'socket.io-client'

class HomePage extends React.Component {
    // subscribeIo() {
    //     this.s = io(`ws://${window.location.host}`) // eslint-disable-line no-undef
    //     this.s.on('connect', () => { console.log('connection open') })
    //
    //     // s.on('disconnect', () => { console.log('disconnect') })
    //     // s.on('tradechange', (data) => { console.log(data) })
    // }

    // componentDidMount() {
    //     const sidebarWidth = this.sidebar.offsetWidth
    //     if (this.state.sidebarWidth !== sidebarWidth) {
    //         this.setState({ sidebarWidth })
    //     }
    // }

    render() {
        const style = {
            height: `calc(100vh - ${this.props.headerHeight}px)`
        }
        return (
            <div>
                <Sidebar.Pushable as='div' style={style}>
                    <SymbolSidebar symbols={this.props.symbols} />
                    <Sidebar.Pusher>
                        <Grid padded stackable>
                            <Grid.Column width={16}>
                                <Container fluid textAlign='center'>
                                    <Icon color='green' name='check circle' size='massive' class='pulsate' />
                                    <Header as='h2' color='green'>No issues</Header>
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
    symbols: React.PropTypes.array, // eslint-disable-line
}

HomePage.defaultProps = {
    headerHeight: 0,
    symbols: [],
}

export default connect(
    state => ({
        symbols: state.db.symbols,
        headerHeight: state.ui.headerHeight,
    }),
)(HomePage)
