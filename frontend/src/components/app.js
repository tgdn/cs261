import React from 'react'
import { connect } from 'react-redux'
import { Container, Label } from 'semantic-ui-react'

import Header from './header'
import Trade from './trade'

// const Horizon = require('@horizon/client')
//
// const horizon = Horizon({ secure: false })
// const chat = horizon('messages')

class App extends React.Component {
    constructor(props) {
        super(props)
        this.state = {
            sectors: []
        }
    }

    componentDidMount() {
        this.subscribe()
    }

    subscribe() {
        this.props.sectors
            .order('name')
            .watch()
            .subscribe((sectors) => {
                this.setState({
                    sectors
                })
            })
    }

    render() {
        return (
            <div>
                <Header />
                <Container>
                    {this.state.sectors.map(sector => (
                        <Label as='span' key={sector.name} tag>
                            {sector.name}
                        </Label>
                    ))}
                </Container>
            </div>
        )
    }
}

export default connect(
    state => ({
        trades: state.db.trades,
        sectors: state.db.sectors,
    }),
)(App)
