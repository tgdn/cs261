import React from 'react'
// import { Link } from 'react-router'
import { Menu } from 'semantic-ui-react'

import AnalysisModal from './analysismodal'

export default function MainHeader() {
    return (
        <Menu inverted stackable borderless fixed='top' size='huge' className='mainmenu'>
            <Menu.Item header>Purple</Menu.Item>
            <Menu.Menu position='right'>
                <Menu.Item>
                    <AnalysisModal />
                </Menu.Item>
            </Menu.Menu>
        </Menu>
    )

    // return (
    //     <div class="luxbar luxbar-fixed">
    //         <input
    //             type="checkbox"
    //             id="luxbar-checkbox"
    //         />
    //         <div class="luxbar-menu luxbar-menu-dark luxbar-menu-black-force">
    //             <ul class="luxbar-navigation">
    //                 <li class="luxbar-header">
    //                     <Link
    //                         class="luxbar-brand"
    //                         to="/"
    //                     >
    //                         Purple
    //                     </Link>
    //                     <label
    //                         class="luxbar-hamburger luxbar-hamburger-spin"
    //                         htmlFor="luxbar-checkbox"
    //                     >
    //                         <span />
    //                     </label>
    //                 </li>
    //                 {/*
    //                 <li class="luxbar-item active"><a href="#">Home</a></li>
    //                 */}
    //             </ul>
    //         </div>
    //     </div>
    // )
}
