import React from 'react'

export default (props) => (
    <div class="luxbar luxbar-fixed">
        <input
            type="checkbox"
            id="luxbar-checkbox"
        />
        <div class="luxbar-menu luxbar-menu-dark">
            <ul class="luxbar-navigation">
                <li class="luxbar-header">
                    <a class="luxbar-brand" href="#">
                        Purple
                    </a>
                    <label
                        class="luxbar-hamburger luxbar-hamburger-spin"
                        for="luxbar-checkbox"
                    >
                        <span></span>
                    </label>
                </li>
                <li class="luxbar-item active"><a href="#">Home</a></li>
            </ul>
        </div>
    </div>
)
