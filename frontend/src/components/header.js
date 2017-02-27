import React from 'react'
// import { Link } from 'react-router'
import { Menu } from 'semantic-ui-react'

import SettingsModal from './settingsmodal'

export default function MainHeader() {
    return (
        <Menu inverted stackable borderless fixed='top' size='huge' className='mainmenu' id='mainmenu'>
            <Menu.Item header>Purple</Menu.Item>
            <Menu.Menu position='right'>
                <Menu.Item>
                    <SettingsModal />
                </Menu.Item>
            </Menu.Menu>
        </Menu>
    )
}
