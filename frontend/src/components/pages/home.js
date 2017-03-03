import React from 'react'
import { connect } from 'react-redux'
import { Link } from 'react-router'
import {
    Container,
    Grid,
    Icon,
    Header,
    Menu,
    Sidebar,
    Segment
} from 'semantic-ui-react'

import groupBy from 'lodash/groupBy'
import map from 'lodash/map'

import io from 'socket.io-client'

class HomePage extends React.Component {
    // subscribeIo() {
    //     this.s = io(`ws://${window.location.host}`) // eslint-disable-line no-undef
    //     this.s.on('connect', () => { console.log('connection open') })
    //
    //     // s.on('disconnect', () => { console.log('disconnect') })
    //     // s.on('tradechange', (data) => { console.log(data) })
    // }

    render() {
        const style = {
            height: `calc(100vh - ${this.props.headerHeight}px)`
        }
        return (
            <div>
                <Sidebar.Pushable as='div' style={style}>
                    <Sidebar as={Menu} animation='push' width='thin' visible icon='labeled' vertical inverted>
                        {map(this.props.symbols, symbol => (
                            <Menu.Item key={symbol.name} name={symbol.name} />
                        ))}
                    </Sidebar>
                    <Sidebar.Pusher>
                        <Segment basic>
                            <Header as='h3'>HEllo</Header>
                        </Segment>
                    </Sidebar.Pusher>
                </Sidebar.Pushable>
            </div>
        )
        //
        //         <Grid padded stackable>
        //             <Grid.Column width={4}>
        //                 <Menu vertical fluid text>
        //                     {/*map(this.props.symbols, symbol => (
        //                         <Menu.Item key={symbol.name} name={symbol.name} />
        //                     ))*/}
        //                 </Menu>
        //             </Grid.Column>
        //             <Grid.Column width={12}>
        //                 <Container textAlign='center'>
        //                     <Icon color='green' name='check circle' size='massive' class='pulsate' />
        //                     <Header as='h2' color='green'>No issues</Header>
        //                 </Container>
        //             </Grid.Column>
        //         </Grid>
        //     </div>
        // )
    }
}

HomePage.propTypes = {
    symbols: React.PropTypes.array, // eslint-disable-line
}

HomePage.defaultProps = {
    symbols: [],
}

export default connect(
    state => ({
        symbols: state.db.symbols,
        headerHeight: state.ui.headerHeight,
    }),
)(HomePage)
