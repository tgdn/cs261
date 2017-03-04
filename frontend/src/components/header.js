import React from 'react'
import { connect } from 'react-redux'
import { Link } from 'react-router'
import { Menu, Loader } from 'semantic-ui-react'

import SettingsModal from './settingsmodal'

const MainHeader = ({ tasks }) => {
    return (
        <Menu inverted stackable borderless fixed='top' size='huge' className='mainmenu' id='mainmenu'>
            <Menu.Item header as={Link} to='/'>Purple</Menu.Item>
            <Menu.Menu position='right'>
                <Menu.Item>
                    {tasks.length !== 0 && (
                        <small>
                            <Loader active inverted inline indeterminate size='small' />
                            <span style={{ marginLeft: '10px' }}>analysing</span>
                        </small>
                    )}
                </Menu.Item>
                <Menu.Item>
                    <SettingsModal />
                </Menu.Item>
            </Menu.Menu>
        </Menu>
    )
}

export default connect(
    state => ({
        tasks: state.tasks.tasks,
    }),
)(MainHeader)
