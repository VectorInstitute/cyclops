// Javascript code for buttons in the model card template


function setActiveButton() {
  const buttons = document.querySelectorAll('#contents li');
  const sections = document.querySelectorAll('.card');

  for (let i = 0; i < sections.length; i++) {
    const section = sections[i];
    if (isInView(section)) {
      // find button from buttons with same id as section with "_button" appended
      for (let j = 0; j < buttons.length; j++) {
        const button = buttons[j];
        if (button.id == section.id + '_button') {
          button.classList.add('active');
        } else {
          button.classList.remove('active');
        }
      }
    }
  }
}


function setCollapseButton() {
  const collapsible = document.getElementsByClassName('collapsible');

  const subcards = document.getElementsByClassName('subcard');
  // set subcards to display: none for all subcards
  for (let j = 0; j < subcards.length; j++) {
    subcards[j].style.display = 'none';
  }

  for (let i = 0; i < collapsible.length; i++) {
    collapsible[i].addEventListener('click', function() {
      const arrow = collapsible[i];
      if (arrow.classList.contains('down-arrow')) {
        arrow.classList.add('right-arrow');
        arrow.classList.remove('down-arrow');
      } else {
        arrow.classList.add('down-arrow');
        arrow.classList.remove('right-arrow');
      }

      const card = this.closest('.card');
      const subcards = card.getElementsByClassName('subcard');

      for (let j = 0; j < subcards.length; j++) {
        subcards[j].style.display = subcards[j].style.display === 'block' ? 'none' : 'block';
      }

      const collapsible_bar = card.getElementsByClassName('collapsible-bar')[0];
      collapsible_bar.style.display = collapsible_bar.style.display === 'block' ? 'none' : 'block';
    });
  }
}
