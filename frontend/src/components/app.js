import React from 'react'
import { Container, Input, Menu, Button, Segment } from 'semantic-ui-react'

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
                        <Menu pointing>
                            <Menu.Item name='home' active />
                            <Menu.Item name='tab2' />
                            <Menu.Item name='tab3' />
                            <Menu.Menu position='right'>
                                <Menu.Item>
                                    <Input icon='search' placeholder='Search...' />
                                </Menu.Item>
                            </Menu.Menu>
                        </Menu>
                    <Segment>
                        <Button basic>Standard</Button>
                        <Button basic color='red'>Red</Button>
                        <Button basic color='orange'>Orange</Button>
                        <Button basic color='yellow'>Yellow</Button>
                        <Button basic color='olive'>Olive</Button>
                        <Button basic color='green'>Green</Button>
                        <Button basic color='teal'>Teal</Button>
                        <Button basic color='blue'>Blue</Button>
                        <Button basic color='violet'>Violet</Button>
                        <Button basic color='purple'>Purple</Button>
                        <Button basic color='pink'>Pink</Button>
                        <Button basic color='brown'>Brown</Button>
                        <Button basic color='grey'>Grey</Button>
                        <Button basic color='black'>Black</Button>
                    </Segment>
                </Container>
            </div>
        )
    }
}

export default App
