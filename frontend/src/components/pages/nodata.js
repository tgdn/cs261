import React from 'react'
import {
    Container,
    Grid,
    Icon,
    Header,
    Sidebar,
    Button,
} from 'semantic-ui-react'

const openSettings = () => {
    /* simply click settings button */
    const settingsBtn = document.getElementById('settings-btn') // eslint-disable-line no-undef
    if (settingsBtn != null) {
        settingsBtn.click()
    }
}

export default () => (
    <Grid padded stackable>
        <Grid.Column width={16}>
            <Container fluid textAlign='center'>
                <div>
                    <Icon color='grey' name='home' size='massive' />
                    <Header as='h2' color='black'>
                        There is no data to work with yet
                    </Header>
                    <Header as='h5' color='grey'>
                        Start analysing a live stream or upload a CSV file
                        to get started
                    </Header>
                    <Button secondary onClick={openSettings}>
                        Open settings
                    </Button>
                </div>
            </Container>
        </Grid.Column>
    </Grid>
)
