import React from 'react'
import { connect } from 'react-redux'

import HeaderNav from './header'

class App extends React.Component {
    render() {
        return (
            <div>
                <HeaderNav />
                {this.props.children}
            </div>
        )
    }
}

export default connect(
    state => ({
        trades: state.db.trades,
    }),
)(App)
