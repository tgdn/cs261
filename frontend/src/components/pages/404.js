import React from 'react'
import {
    Container,
    Icon,
    Header,
} from 'semantic-ui-react'

export default function PageDoesNotExist() {
    return (
        <Container>
            <Header as='h1' icon textAlign='center'>
                <Icon name='treatment' circular />
                <Header.Content>
                    {"Ehm... This page doesn't exist"}
                </Header.Content>
            </Header>
        </Container>
    )
}
