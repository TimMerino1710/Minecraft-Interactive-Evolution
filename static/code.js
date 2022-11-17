// Javascript code for the Minecraft Interactive House Evolver
// Written by Milk

var CUR_MODE = "evolve";

var HOUSE_PNGS = [];
var HOUSE_GIFS = [];
var HOUSE_ARR = [];
var HOUSE_Z = {};

var def_img = "static/tmp/spin_block.gif"


//initializing function executed when the page loads
function init(json_dat){
    console.log("code.js init");

    // document.getElementById("debug").innerHTML = ((json_dat == "" || json_dat == undefined) ? "nothing" : "something!");
    // console.log(json_dat);

    //import the data recieved from flask if available
    if(json_dat != "" && json_dat != undefined){
        document.getElementById("load_overlay").style.display = "block";
        console.log("got valid data");

        HOUSE_ARR = JSON.parse(json_dat.house_arr);
        HOUSE_Z = JSON.parse(json_dat.z_set);

        //render the houses
        makeHouseConfig();
        importSubTextures();

        //wait for the textures to load 
        setTimeout(function(){
            loadHouses();
            //remove the generator window
            document.getElementById("load_overlay").style.display = "none";
        }, 1000);
        
    //otherwise default to the start screen
    }else{
        console.log("got no data");
        //if no data is recieved, notify the user to hit the generate or reset button
        setInterval(flashBtn, 700);

        //change the generate button to another version of the reset button (same functionality)
        document.getElementById("genBtn").onclick = function(){subForm(document.getElementById("resetBtn"))};

        //set the src image to default
        let houseItems = document.getElementsByClassName("houseItem");
        for(let i = 0; i < houseItems.length; i++){
            houseItems[i].src = def_img;
        }
    }
    
    setMode("evolve");
    
}

// Set the png and gif renders of the house
function loadHouses(){
    //save the src paths for the png and gif versions of the house
    HOUSE_PNGS = [];
    HOUSE_GIFS = [];

    //create the PNGs
    for(let i = 0; i < HOUSE_ARR.length; i++){
        HOUSE_PNGS.push(renderPNGdat(HOUSE_ARR[i]));
    }

    //add the src images + backup in case of error
    let houseItems = document.getElementsByClassName("houseItem");
    for(let i = 0; i < houseItems.length; i++){
        houseItems[i].src = HOUSE_PNGS[i];
        houseItems[i].onerror = function(){houseItems[i].src = def_img};
        unpreviewHouse(i);
    }

    //create the GIFs (later)
    setTimeout(function(){
        document.getElementById("debug").innerHTML = "Rendering gifs..."

        //make the GIFs
        for(let i = 0; i < HOUSE_ARR.length; i++){
            HOUSE_GIFS.push(renderGIFdat(HOUSE_ARR[i]));
        }

        //add event listeners to the house items
        for(let i = 0; i < houseItems.length; i++){
            houseItems[i].onmouseenter = function(){previewHouse(i)};
            houseItems[i].onmouseleave = function(){unpreviewHouse(i)};
        }

        document.getElementById("debug").innerHTML = ""
    },1);

    
}

// set the render configuation
function makeHouseConfig(){
    CONFIG.CUR_TEXTURE_LIST = ["air","stonebrick","dirt","planks_oak","sand","iron_bars","glass","iron_block","log_oak","wool_colored_red","stone_slab_side"]
    // texture_set = ['air','stone','dirt','planks_oak','lapis_block','sand','leaves','glass','red_flower','stone_slab_side','wool_colored_red','iron_fence']

    CONFIG.ANGLE = 230
    CONFIG.RADIUS = 17
    CONFIG.use_struct_center = true;
}

//preview the house's rotational gif
function previewHouse(houseIndex){
    document.getElementById("house" + houseIndex).src = HOUSE_GIFS[houseIndex];
}

//set back to the png version of the house
function unpreviewHouse(houseIndex){
    document.getElementById("house" + houseIndex).src = HOUSE_PNGS[houseIndex];
}

//activated when a house is clicked on
function clickHouse(houseIndex){
    if (CUR_MODE == "evolve"){
        // Evolve the house
    }else if(CUR_MODE == "save"){
        // Save the house and output its 3d array
        document.getElementById("houseArr").style.display = "block";
        document.getElementById("houseOut").innerHTML = JSON.stringify(HOUSE_ARR[houseIndex]);
    }
}

//set the current mode of the interactive evolver
function setMode(mode){
    if(mode == "evolve"){
        CUR_MODE = "evolve";
        document.getElementById("houseArr").style.display = "none";  //hide the house array
        document.getElementById("debug").innerHTML = "";            //clear the debug text
        document.getElementById("evolveBtn").classList.add("active");  //highlight the evolve button
        document.getElementById("saveBtn").classList.remove("active");  //unhighlight the save button
    }else if(mode == "save"){
        CUR_MODE = "save";
        
        document.getElementById("saveBtn").classList.add("active");  //highlight the save button
        document.getElementById("evolveBtn").classList.remove("active");  //unhighlight the evolve button
    }
}

//toggle the color of the generate/reset button to stand out to the user
let FLASH_COLOR = "#dedede";
function flashBtn(){
    if(FLASH_COLOR == "#dedede")
        FLASH_COLOR = "#FFE400";
    else
        FLASH_COLOR = "#dedede";

    document.getElementById("genBtn").style.backgroundColor = FLASH_COLOR;
    document.getElementById("resetBtn").style.backgroundColor = FLASH_COLOR;

}

//submit a button's form
function subForm(b){
    document.getElementById("load_overlay").style.display = "block";
    b.parentElement.submit();
}

//copy the house array to the clipboard
function copyArr(){
    let copyText = document.getElementById("houseOut");

    // Select the text field
    copyText.select();
    copyText.setSelectionRange(0, 99999); // For mobile devices

    // Copy the text inside the text field
    navigator.clipboard.writeText(copyText.value);

    document.getElementById("debug").innerHTML = "Copied!";
}