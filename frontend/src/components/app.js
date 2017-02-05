import React from 'react'
import { connect } from 'react-redux'
import { Container } from 'semantic-ui-react'

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
            trades: []
        }
    }

    componentDidMount() {
        this.subscribe()
    }

    subscribe() {
        this.props.trades
            .order('time', 'descending')
            .limit(20)
            .watch()
            .subscribe((trades) => {
                this.setState({
                    trades
                })
            })
    }

    render() {
        return (
            <div>
                <Header />
                <Container>
                    {this.state.trades.map(trade => (
                        <Trade key={trade.id} trade={trade} />
                    ))}
                </Container>
            </div>
        )
    }
}

export default connect(
    state => ({
        trades: state.db.trades,
    }),
)(App)
