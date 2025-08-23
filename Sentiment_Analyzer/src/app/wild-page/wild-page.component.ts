import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';

@Component({
  selector: 'app-wild-page',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './wild-page.component.html',
  styleUrl: './wild-page.component.css',
})
export class WildPageComponent {}
