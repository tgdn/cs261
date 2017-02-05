import React from 'react'
import { Container } from 'semantic-ui-react'

import Header from './header'

// const Horizon = require('@horizon/client')
//
// const horizon = Horizon({ secure: false })
// const chat = horizon('messages')

class App extends React.Component {
    render() {
        return (
            <div>
                <Header />
                <Container>
                </Container>
            </div>
        )
    }
}

export default App
